import argparse
import os


def model_info(model, log_dir):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())
    # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    model_summary = 'Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g)
    print(model_summary)
    parameters_path = os.path.join(log_dir, "parameters.txt")
    with open(parameters_path, "a") as parameters:
        parameters.write(str(model_summary))


def write_Args(log_dir, args):
    opts_path = os.path.join(log_dir, "opts.txt")
    with open(opts_path, "a") as opts_args:
        opts_args.write(str("model: " + args.model))
        opts_args.write("\n")
        opts_args.write(str("epochs: " + str(args.epochs)))
        opts_args.write("\n")
        opts_args.write(str("batch_size: " + str(args.batch_size)))
        opts_args.write("\n")
        opts_args.write(str("loss: " + args.loss))
        opts_args.write("\n")
        opts_args.write(str("loss_a: " + str(args.loss_a)))
        opts_args.write("\n")
        opts_args.write(str("num_workers: " + str(args.num_workers)))
        opts_args.write("\n")
        opts_args.write(str("outdir: " + args.outdir))
        opts_args.write("\n")
        opts_args.write(str("lr_step: " + str(args.lr_step)))
        opts_args.write("\n")
        opts_args.write(str("lr_gamma: " + str(args.lr_gamma)))
        opts_args.write("\n")
        opts_args.write(str("num_class: " + str(args.num_class)))
        opts_args.write("\n")
        opts_args.write(str("inputs_class: " + str(args.inputs_class)))
        opts_args.write("\n")
        opts_args.write(str("datadir: " + str(args.datadir)))
        opts_args.write("\n")
        opts_args.write(str("accu_steps: " + str(args.accu_steps)))
        opts_args.write("\n")