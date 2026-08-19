import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.amp import GradScaler
import gc
from contextlib import contextmanager
import time
import concurrent.futures

@contextmanager
def memory_manager():
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


@torch.inference_mode()
def Eval(net, eval_dataloader, device, criterion, epoch, total_step, best_evalloss, save_path, event_writer, logger) -> float:
    """
    Evaluate the model on the eval dataset.

    Args:
        net: The neural network model
        eval_dataloader: DataLoader for eval data
        device: Device to run the model on (cuda/cpu)
        criterion: Loss function
        epoch: Current epoch number
        total_step: Total training steps so far
        best_evalloss: Best loss achieved so far
        save_path: Path to save model checkpoints
        event_writer: TensorBoard writer
        logger: Logger instance

    Returns:
        tuple: (updated best_evalloss, average eval loss)
    """

    net.eval()
    eval_loss = AverageMeter()

    with memory_manager():
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluatinging", unit="batch", leave=False)):
                # Efficient data transfer
                inp = batch['img_ds'].to(device, non_blocking=True)
                target = batch['img_orig'].to(device, non_blocking=True)

                # Forward pass
                outp = net(inp)
                loss = criterion(target=target, output=outp['output'])
                eval_loss.update(loss.item(), inp.size(0))

                del inp, target, outp, loss

        if event_writer:
            try:
                logger.info(f"Logging eval data: epoch={epoch}, step={total_step}")
                safe_tensorboard_write(logger, event_writer, event_writer.add_scalar, 5, "Eval instant loss_avg", eval_loss.avg, total_step)
            except Exception as e:
                logger.info(f'Visualization error at step {total_step}: {e}')

        # Save best model
        if eval_loss.avg < best_evalloss:
            best_evalloss = eval_loss.avg
            save_path = Path(save_path) / 'Models' / 'Model_best_evalLoss.pth.tar'
            torch.save({
                'epoch': epoch,
                'total_step': total_step,
                'state_dict': net.state_dict(),
                'eval_loss': best_evalloss
            }, save_path)
            logger.info(f"New best eval loss: {best_evalloss:.6f}")

    net.train()
    return best_evalloss, eval_loss.avg

def Train(net, train_dataloader, eval_dataloader, epoch, device, criterion, optimizer, lr_scheduler, eval_freq,
          total_step, total_epochs, best_evalloss, save_path, event_writer, logger, clip_max_norm=1.0) -> tuple[int, float]:
    """
    Train the model for one epoch.

    Args:
        net: The neural network model
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for eval data
        epoch: Current epoch number
        device: Device to run the model on (cuda/cpu)
        criterion: Loss function
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
        eval_freq: How often to run evaluation
        total_step: Total training steps so far
        total_epochs: Total number of epochs to train
        best_evalloss: Best loss achieved so far
        save_path: Path to save model checkpoints
        event_writer: TensorBoard writer
        logger: Logger instance
        clip_max_norm: Maximum norm for gradient clipping

    Returns:
        tuple: (updated total_step, updated best_evalloss)
    """

    net.train()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    epoch_loss = AverageMeter()
    epoch_eval_loss = AverageMeter()
    scaler = GradScaler(device.type)

    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{total_epochs}', unit='batch', leave=False)

    for batch_idx, batch in enumerate(pbar):
        with memory_manager():
            # Efficient data transfer
            inp = batch['img_ds'].to(device, non_blocking=True)
            target = batch['img_orig'].to(device, non_blocking=True)

            # Forward pass with automatic mixed precision
            optimizer.zero_grad(set_to_none=True)

            # with autocast(device.type):
            outp = net(inp)
            loss = criterion(target=target, output=outp['output'])

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            if clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_max_norm)

            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            epoch_loss.update(loss.item(), inp.size(0))
            pbar.set_postfix(avg_epoch_loss=f"{epoch_loss.avg:.4f}")

            #Log training metrics
            if event_writer is not None:
                try:
                    safe_tensorboard_write(logger, event_writer, event_writer.add_scalar, 5, "Train loss", loss.item(), total_step)
                except Exception as e:
                    logger.info(f'problem writing the eventwriter in train step {total_step} because - {e}')

            if (total_step % eval_freq) == 0:
                best_evalloss, eval_loss = Eval(net, eval_dataloader, device, criterion, epoch, total_step, best_evalloss,
                                     save_path, event_writer, logger)
                epoch_eval_loss.update(eval_loss, eval_dataloader.batchsize*len(eval_dataloader))
                if event_writer:
                    try:
                        logger.info(f"Logging epoch eval data: epoch={epoch}, step={total_step}")
                        safe_tensorboard_write(logger, event_writer, event_writer.add_scalar, 5, "Eval loss_avg",
                                               epoch_eval_loss.avg, total_step)
                    except Exception as e:
                        logger.info(f'Visualization error at step {total_step}: {e}')

            del inp, target, outp, loss
            total_step += 1
            pbar.update(1)

    lr_scheduler.step()
    logger.info(f"Epoch {epoch + 1}: avg_loss={epoch_loss.avg:.6f}")
    return total_step, best_evalloss


def profile_tensorboard_write(logger, event_writer, log_func, *args, threshold=1, **kwargs):
    """
    Measures execution time of TensorBoard writing.

    Args:
        event_writer: TensorBoard event writer instance.
        log_func: TensorBoard logging function (e.g., add_scalar, add_image).
        threshold: Time (seconds) above which a warning is printed.
        *args, **kwargs: Arguments to pass to the log function.
    """
    if event_writer is None:
        return  # Skip logging if TensorBoard is not initialized

    start_time = time.time()
    try:
        log_func(*args, **kwargs)
    except Exception as e:
        logger.info(f"⚠️ TensorBoard logging error: {e}")
        return
    elapsed_time = time.time() - start_time

    if elapsed_time > threshold:
        logger.info(f"⚠️ WARNING: TensorBoard logging took {elapsed_time:.2f}s (exceeds {threshold}s)")


def safe_tensorboard_write(logger, event_writer, log_func, timeout=1, *args, **kwargs):
    """
    Writes to TensorBoard with a timeout.

    Args:
        event_writer: The TensorBoard event writer instance.
        log_func: The function to execute (e.g., add_scalar, add_image).
        timeout: Maximum time (in seconds) before skipping the logging.
        *args, **kwargs: Arguments to pass to the log function.
    """
    if event_writer is None:
        return  # Skip if TensorBoard is not initialized

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(log_func, *args, **kwargs)
            future.result(timeout=timeout)  # Raise TimeoutError if it takes too long
    except concurrent.futures.TimeoutError:
        logger.info(f"⚠️ TensorBoard logging skipped due to timeout ({timeout} seconds)")
    except Exception as e:
        logger.info(f"⚠️ TensorBoard logging error: {e}")
