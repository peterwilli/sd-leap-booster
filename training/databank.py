import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

def make_bank(tensor, plot = False):
    t = torch.linspace(0, 1, tensor.shape[0])
    coeffs = natural_cubic_spline_coeffs(t, tensor)
    spline = NaturalCubicSpline(coeffs)

    del tensor
    
    def bank_fn(pos_args):
        result = spline.evaluate(pos_args)
        result = torch.sum(result, dim=1)
        return result / pos_args.shape[1]
    return bank_fn
    
def dataloader_to_bank(loader):
    full_data = None
    for x, y, _ in loader:
        if full_data is None:
            full_data = y
        else:
            full_data = torch.cat((full_data, y), dim=0)
    print("full_data", full_data.shape)
    # to_add, _ = torch.sort(data, dim=1)
    bank_fn = make_bank(full_data, False)
    return bank_fn