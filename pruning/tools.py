import numpy as np


def compute_compression2(model):
    present_weights = 0
    total_weights = 0

    for v in model.children():
        if hasattr(v, "weight"):
            arr = v.weight.cpu().detach().numpy()
            arrs = arr[np.nonzero(v.mask.numpy())]
            total_weights += arr.shape[0]*arr.shape[1]
            present_weights += arrs.shape[0]
            # print(v.weight[0])

    tmp = (present_weights/total_weights)*100
    if tmp == 0:
        return 0
    return 100.0/(tmp)


def compute_compression(tuples):
    t = 0
    nz = 0
    zeroed = 0

    for layer, name in tuples:
        found = False
        for a, mask in list(layer.named_buffers()):
            if name+"_mask" == a:
                t += np.prod(mask.shape)
                nz += len(np.nonzero(mask.detach().cpu().numpy())[0])
                zeroed += t-nz
        if found:
            tensor = layer.__getattr__(name)
            t += np.prod(tensor.shape)
            nz += len(np.nonzero(tensor.detach().cpu().numpy())[0])
            zeroed += t-nz

    return int(zeroed), t/(nz+1e-8)


def get_comp_info(tuples):
    t = 0
    nz = 0

    for layer, name in tuples:
        found = False
        for a, mask in list(layer.named_buffers()):
            if name+"_mask" == a:
                t += np.prod(mask.shape)
                nz += len(np.nonzero(mask.detach().cpu().numpy())[0])
        if found:
            tensor = layer.__getattr__(name)
            t += np.prod(tensor.shape)
            nz += len(np.nonzero(tensor.detach().cpu().numpy())[0])

    return t, nz


def tuples_retriever(model, embeddings=False, bias=False):
    out = []
    if embeddings or (not embeddings and "embedding" not in model.__class__.__name__.lower()):
        children = False
        for child in model.children():
            out = out + tuples_retriever(child, embeddings, bias)
            children = True

        if not children:
            for name, _ in model.named_parameters():
                if "weight" in name or (bias and "bias" in name):
                    out.append((model, name))

    return out


def retrieve_tuples(model, embeddings=False, bias=False, last_layer=False):
    tuples = tuples_retriever(model, embeddings=embeddings, bias=bias)
    if not last_layer:
        tuples = tuples[:-1]
        if bias:
            tuples = tuples[:-1]
    return tuples


def tot_weights(tuples):
    t = 0
    for layer, name in tuples:
        found = False
        for a, mask in list(layer.named_buffers()):
            if name+"_mask" == a:
                t += np.prod(mask.shape)
                found = True
        if not found:
            tensor = layer.__getattr__(name)
            t += np.prod(tensor.shape)
    return t
