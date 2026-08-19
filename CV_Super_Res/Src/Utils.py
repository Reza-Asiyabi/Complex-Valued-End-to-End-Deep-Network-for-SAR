def setup_finetuning(net, finetuning_strategy):
    if not finetuning_strategy:
        return net
    else:
        for param in net.parameters():
            param.requires_grad = False
        if 'En' in finetuning_strategy:
            for param in net.entry_block.parameters():
                param.requires_grad = True
        if 'Fe' in finetuning_strategy:
            try:
                for param in net.feature_block.parameters():
                    param.requires_grad = True
            except:
                for param in net.sepconv_blocks.parameters():
                    param.requires_grad = True
        if 'Pr' in finetuning_strategy:
            try:
                for param in net.prediction_block.parameters():
                    param.requires_grad = True
            except:
                for param in net.predictions.parameters():
                    param.requires_grad = True
        if 'Va' in finetuning_strategy:
            try:
                for param in net.variances_block.parameters():
                    param.requires_grad = True
            except:
                for param in net.variances.parameters():
                    param.requires_grad = True
        return net
