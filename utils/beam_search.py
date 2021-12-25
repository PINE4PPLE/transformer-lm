import torch
EOS_ID = 1
INF = 1. * 1e7

def _merge_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.

    Args:
        tensor: Tensor to reshape of shape [A, B, ...]

    Returns:
        Reshaped tensor of shape [A*B, ...]
    """
    shape = list(tensor.shape)
    shape[0] *= shape[1]  # batch -> batch * beam_size
    shape.pop(1)  # Remove beam dim
    return torch.reshape(tensor, shape)


def _unmerge_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].

    Args:
        tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
        batch_size: Tensor, original batch size.
        beam_size: int, original beam size.

    Returns:
        Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = list(tensor.shape)
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return torch.reshape(tensor, new_shape)


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.

    Args:
        tensor: tensor to tile [batch_size, ...]
        beam_size: How much to tile the tensor by.

    Returns:
        Tiled tensor [batch_size, beam_size, ...]
    """
    tensor = torch.unsqueeze(tensor, 1)
    tile_dims = [1] * len(tensor.shape)
    tile_dims[1] = beam_size

    return tensor.repeat(tile_dims)

def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coordinate that contains the batch index for gathers.

    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.

    Args:
        batch_size: Batch size
        beam_size: Size of the beam.
    Returns:
        batch_pos: [batch_size, beam_size] tensor of ids
    """
    batch_pos = torch.arange(batch_size * beam_size)// beam_size
    batch_pos = torch.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos
def compute_seq_length(topk,seq_len):
    tensor = torch.unsqueeze(topk, -1)
    tile_dims = [1] * len(tensor.shape)
    tile_dims[-1] = seq_len

    res = tensor.repeat(tile_dims)
    return res

def compute_topk_scores_and_seq(sequences,
                                scores,
                                scores_to_gather,
                                flags,
                                beam_size,
                                batch_size
                                ):
    """Given sequences and scores, will gather the top k=beam size sequences.

    This function is used to grow alive, and finished. It takes sequences,
    scores, and flags, and returns the top k from sequences, scores_to_gather,
    and flags based on the values in scores.

    This method permits easy introspection using tfdbg.  It adds three named ops
    that are prefixed by `prefix`:
        - _topk_seq: the tensor for topk_seq returned by this method.
        - _topk_flags: the tensor for topk_finished_flags returned by this method.
        - _topk_scores: the tensor for tokp_gathered_scores returned by this method.

    Args:
        sequences: Tensor of sequences that we need to gather from.
        [batch_size, beam_size, seq_length]
        scores: Tensor of scores for each sequence in sequences.
        [batch_size, beam_size]. We will use these to compute the topk.
        scores_to_gather: Tensor of scores for each sequence in sequences.
        [batch_size, beam_size]. We will return the gathered scores from here.
        Scores to gather is different from scores because for grow_alive, we will
        need to return log_probs, while for grow_finished, we will need to return
        the length penalized scores.
        flags: Tensor of bools for sequences that say whether a sequence has reached
        EOS or not
        beam_size: int
        batch_size: int

    Returns:
        Tuple of
        (topk_seq [batch_size, beam_size, decode_length],
        topk_gathered_scores [batch_size, beam_size],
        topk_finished_flags[batch_size, beam_size])
    """

    _, topk_indexes = torch.top_k(scores, k=beam_size) #[batch_size, beam_size ]
    seq_indexes = compute_seq_length(topk_indexes, sequences.shape[-1])
    # The next three steps are to create coordinates for tf.gather_nd to pull
    # out the topk sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j coordinate
    # needed for the gather
    #batch_pos = compute_batch_indices(batch_size, beam_size)

    # top coordinates will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where the
    # last dimension contains the i,j gathering coordinates.
    #top_coordinates = torch.stack((batch_pos, topk_indexes), axis=2)

    # Gather up the highest scoring sequences.  For each operation added, give
    # it a concrete name to simplify observing these operations with tfdbg.
    # Clients can capture these tensors by watching these node names.
    #topk_seq = torch.gather(sequences, top_coordinates)
    #topk_flags = torch.gather(flags, top_coordinates)
    #topk_gathered_scores = torch.gather(scores_to_gather, top_coordinates)
    topk_seq = torch.gather(sequences,1, seq_indexes)
    topk_flags = torch.gather(flags, 1, topk_indexes)
    topk_gathered_scores = torch.gather(scores_to_gather,1, topk_indexes)

    return topk_seq, topk_gathered_scores, topk_flags

def beam_search(symbols_to_logits_fn,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True,
                use_tpu=False,
                use_top_k_with_unique=True):