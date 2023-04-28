import torch as th

def w_distance_quant_cat(
    quant_dist: th.Tensor,
    cat_dist: th.Tensor,
    cat_zs: th.Tensor,
    cdf_dim: int,
    p: int,
):
    # expand to combinatorial
    # dim = c
    # (a,b,1,c,d) - (a,b,c,1,d) -> (a,b,c,c,d)
    pairwise_delta = th.abs(quant_dist.unsqueeze(cdf_dim-1)-cat_zs.unsqueeze(cdf_dim))
    
    if p==1:
        pairwise_delta = pairwise_delta
    elif p==2:
        pairwise_delta = pairwise_delta**2
    else:
        pairwise_delta = th.pow(pairwise_delta, p)

    int_delta = th.sum(pairwise_delta, dim=cdf_dim+1)
    int_delta = th.mean(th.mean(int_delta, dim=ens_dim), dim=ens_dim)

    return int_delta
