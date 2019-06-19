import torch
import types
import numpy as np

from torch.autograd import Variable, Function

__all__ = ['VanillaGradExplainer', 'GradxInputExplainer', 'SaliencyExplainer', 'IntegrateGradExplainer',
           'DeconvExplainer', 'SmoothGradExplainer', 'GradCAMExplainer', 'get_explainer']


class VanillaGradExplainer(object):
    def __init__(self, model):
        super(VanillaGradExplainer, self).__init__()
        self.model = model

    def _backprop(self, inp, ind):
        inp.requires_grad = True
        if inp.grad is not None:
            inp.grad.zero_()
        if ind.grad is not None:
            ind.grad.zero_()
        self.model.eval()
        self.model.zero_grad()

        output = self.model(inp)
        if ind is None:
            ind = output.max(1)[1]
        grad_out = output.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        output.backward(grad_out)

        return inp.grad

    def explain(self, inp, ind=None, raw_inp=None):
        return self._backprop(inp, ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GradxInputExplainer, self).__init__(model)

    def explain(self, inp, ind=None, raw_inp=None):
        grad = self._backprop(inp, ind)

        return inp * grad


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(SaliencyExplainer, self).__init__(model)

    def explain(self, inp, ind=None, raw_inp=None):
        grad = self._backprop(inp, ind)

        return grad.abs()


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100):
        super(IntegrateGradExplainer, self).__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None, raw_inp=None):
        grad = 0
        inp_data = inp.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / self.steps


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(DeconvExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


class SmoothGradExplainer(object):
    def __init__(self, model, base_explainer=None, stdev_spread=0.15, nsamples=25, magnitude=True):
        self.base_explainer = base_explainer or VanillaGradExplainer(model)
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude

    def explain(self, inp, ind=None, raw_inp=None):
        stdev = self.stdev_spread * (inp.max() - inp.min())
        total_gradients = 0

        for i in range(self.nsamples):
            noise = torch.randn_like(inp) * stdev
            noisy_inp = inp + noise
            noisy_inp.retain_grad()
            grad = self.base_explainer.explain(noisy_inp, ind)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples


# -------------------------------------------------
#                   GradCAM
# -------------------------------------------------
class GradCAMExplainer(VanillaGradExplainer):
    def __init__(self, model, target_layer_name_keys=None, use_inp=False):
        super(GradCAMExplainer, self).__init__(model)
        self.target_layer = self._get_layer(model, target_layer_name_keys)
        self.use_inp = use_inp
        self.intermediate_act = []
        self.intermediate_grad = []
        self._register_forward_backward_hook()

    def _register_forward_backward_hook(self):
        def forward_hook_input(m, i, o):
            self.intermediate_act.append(i[0].data.clone())

        def forward_hook_output(m, i, o):
            self.intermediate_act.append(o.data.clone())

        def backward_hook(m, grad_i, grad_o):
            self.intermediate_grad.append(grad_o[0].data.clone())

        if self.target_layer is not None:
            if self.use_inp:
                self.target_layer.register_forward_hook(forward_hook_input)
            else:
                self.target_layer.register_forward_hook(forward_hook_output)

            self.target_layer.register_backward_hook(backward_hook)

    def _reset_intermediate_lists(self):
        self.intermediate_act = []
        self.intermediate_grad = []

    def explain(self, inp, ind=None, raw_inp=None):
        self._reset_intermediate_lists()

        _ = super(GradCAMExplainer, self)._backprop(inp, ind)

        if len(self.intermediate_grad):
            grad = self.intermediate_grad[0]
            act = self.intermediate_act[0]

            weights = grad.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
            cam = weights * act
            cam = cam.sum(1).unsqueeze(1)

            cam = torch.clamp(cam, min=0)

            return cam
        else:
            return None

    @classmethod
    def _get_layer(cls, model, key_list):
        if key_list is None:
            return None

        a = model
        for key in key_list:
            a = a._modules[key]
        return a


def _get_layer_path(model):
    if model.__class__.__name__ == 'VGG':
        return ['features', '30']  # pool5
    elif model.__class__.__name__ == 'GoogleNet':
        return ['pool5']
    elif model.__class__.__name__ == 'ResNet':
        return ['avgpool']  # layer4
    elif model.__class__.__name__ == 'Inception3':
        return ['Mixed_7c', 'branch_pool']  # ['conv2d_94'], 'mixed10'
    else:  # TODO: guess layer for other networks?
        return None


def get_explainer(explainer_name, model, layer_path=None):
    layer_path = layer_path or _get_layer_path(model)
    if explainer_name == 'gradcam':
        return GradCAMExplainer(model, target_layer_name_keys=layer_path, use_inp=True)
    elif explainer_name == 'vanilla_grad':
        return VanillaGradExplainer(model)
    elif explainer_name == 'grad_x_input':
        return GradxInputExplainer(model)
    elif explainer_name == 'saliency':
        return SaliencyExplainer(model)
    elif explainer_name == 'integrate_grad':
        return IntegrateGradExplainer(model)
    elif explainer_name == 'deconv':
        return DeconvExplainer(model)
    elif explainer_name == 'smooth_grad':
        return SmoothGradExplainer(model)
    else:
        raise ValueError('Explainer {} is not recognized'.format(explainer_name))
