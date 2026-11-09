"""BitFit (Ben Zaken et al. 2021, arXiv 2106.10199): fine-tune only the bias terms.

IDEA: freeze every weight matrix, train only the biases (+ the new task head). That's ~0.1% of the parameters.
Note: Models 4 / 5 have no biases at all (bias=False everywhere), so BitFit only applies to Models 2 / 3.

Torch basics you will need:
  model.named_parameters()     -> yields (name, parameter), e.g. ("enc_rnn.bias_ih_l0", tensor), ("head.bias", ...)
  p.requires_grad = False      -> freeze one parameter
  name.endswith("bias")        -> string test;  "bias_ih" in name  -> substring test
"""
import torch.nn as nn


def apply_bitfit(model, keep_trainable=()):
    """Freeze every parameter except the bias terms (for nn.GRU these are bias_ih_l* / bias_hh_l*).
    keep_trainable: modules to leave fully trainable anyway (e.g. the new classification head).
    Returns the list of bias parameters that are trainable."""
    # Step 1. Loop over model.named_parameters(): a parameter is a bias if its name ends with "bias"
    #         or contains "bias_ih" / "bias_hh" (the GRU's names). Set requires_grad = True for biases, False otherwise,
    #         and collect the biases in a list.
    # Step 2. For every module in keep_trainable: set requires_grad = True on all its parameters
    #         (module.parameters()).
    # Step 3. If the bias list is empty, raise an error (the model has no biases -> BitFit does not apply).
    # Step 4. return the list
    raise NotImplementedError


if __name__ == "__main__":
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    apply_bitfit(m)
    print([n for n, p in m.named_parameters() if p.requires_grad])   # expect ['0.bias', '2.bias']
