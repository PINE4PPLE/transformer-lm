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
                stop_early=True):
    """Beam search with length penalties.

    Requires a function that can take the currently decoded symbols and return
    the logits for the next symbol. The implementation is inspired by
    https://arxiv.org/abs/1609.08144.

    When running, the beam search steps can be visualized by using tfdbg to watch
    the operations generating the output ids for each beam step.  These operations
    have the pattern:
        (alive|finished)_topk_(seq,scores)

    Operations marked `alive` represent the new beam sequences that will be
    processed in the next step.  Operations marked `finished` represent the
    completed beam sequences, which may be padded with 0s if no beams finished.

    Operations marked `seq` store the full beam sequence for the time step.
    Operations marked `scores` store the sequence's final log scores.

    The beam search steps will be processed sequentially in order, so when
    capturing observed from these operations, tensors, clients can make
    assumptions about which step is being recorded.

    WARNING: Assumes 2nd dimension of tensors in `states` and not invariant, this
    means that the shape of the 2nd dimension of these tensors will not be
    available (i.e. set to None) inside symbols_to_logits_fn.

    Args:
        symbols_to_logits_fn: Interface to the model, to provide logits.
            Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size]
        initial_ids: Ids to start off the decoding, this will be the first thing
            handed to symbols_to_logits_fn (after expanding to beam size)
            [batch_size]
        beam_size: Size of the beam.
        decode_length: Number of steps to decode for.
        vocab_size: Size of the vocab, must equal the size of the logits returned by
            symbols_to_logits_fn
        alpha: alpha for length penalty.
        states: dict (possibly nested) of decoding states.
        eos_id: ID for end of sentence.
        stop_early: a boolean - stop once best sequence is provably determined.
        use_tpu: A bool, whether to do beam search on TPU.
        use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
        top_k during TPU beam search.

    Returns:
        Tuple of
        (decoded beams [batch_size, beam_size, decode_length]
        decoding probabilities [batch_size, beam_size])
    """
    batch_size = initial_ids.shape[0]
    initial_log_probs = torch.tensor([[0.] + [-INF] * (beam_size - 1)])
    # Expand to beam_size (batch_size, beam_size)
    alive_log_probs = initial_log_probs.repeat([batch_size, 1])
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    alive_seq = torch.unsqueeze(alive_seq, 2)# (batch_size, beam_size, 1)
    # Finished will keep track of all the sequences that have finished so far
    # Finished log probs will be negative infinity in the beginning
    # finished_flags will keep track of booleans
    finished_seq = torch.zeros(list(alive_seq.shape), torch.int32)
    # Setting the scores of the initial to negative infinity.
    finished_scores = torch.ones([batch_size, beam_size]) * -INF
    finished_flags = torch.zeros([batch_size, beam_size], torch.bool)

    def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                    curr_scores, curr_finished):
        """Given sequences and scores, will gather the top k=beam size sequences.

        Args:
        finished_seq: Current finished sequences.
            [batch_size, beam_size, current_decoded_length]
        finished_scores: scores for each of these sequences.
            [batch_size, beam_size]
        finished_flags: finished bools for each of these sequences.
            [batch_size, beam_size]
        curr_seq: current topk sequence that has been grown by one position.
            [batch_size, beam_size, current_decoded_length]
        curr_scores: scores for each of these sequences. [batch_size, beam_size]
        curr_finished: Finished flags for each of these sequences.
            [batch_size, beam_size]
        Returns:
        Tuple of
            (Topk sequences based on scores,
            log probs of these sequences,
            Finished flags of these sequences)
        """
        
        finished_seq = torch.cat(
                [finished_seq,
                torch.zeros([batch_size, beam_size, 1], torch.int32)], axis=2)

        # Set the scores of the unfinished seq in curr_seq to large negative
        # values
        curr_scores += (1. - curr_finished.to(torch.float)) * -INF
        # concatenating the sequences and scores along beam axis
        curr_finished_seq = torch.cat([finished_seq, curr_seq], 1)
        curr_finished_scores = torch.cat([finished_scores, curr_scores], 1)
        curr_finished_flags = torch.cat([finished_flags, curr_finished], 1)
        return compute_topk_scores_and_seq(
            curr_finished_seq,
            curr_finished_scores,
            curr_finished_scores,
            curr_finished_flags,
            beam_size,
            batch_size
           )
    
    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished):
        """Given sequences and scores, will gather the top k=beam size sequences.

        Args:
        curr_seq: current topk sequence that has been grown by one position.
            [batch_size, beam_size, i+1]
        curr_scores: scores for each of these sequences. [batch_size, beam_size]
        curr_log_probs: log probs for each of these sequences.
            [batch_size, beam_size]
        curr_finished: Finished flags for each of these sequences.
            [batch_size, beam_size]
        Returns:
        Tuple of
            (Topk sequences based on scores,
            log probs of these sequences,
            Finished flags of these sequences)
        """
        # Set the scores of the finished seq in curr_seq to large negative
        # values
        curr_scores += curr_finished.to(torch.float) * -INF
        return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                        curr_finished, beam_size, batch_size)  
    
    def grow_topk(i, alive_seq, alive_log_probs):
        """Inner beam search loop.
        This function takes the current alive sequences, and grows them to topk
        sequences where k = 2*beam. We use 2*beam because, we could have beam_size
        number of sequences that might hit <EOS> and there will be no alive
        sequences to continue. With 2*beam_size, this will not happen. This relies
        on the assumption the vocab size is > beam size. If this is true, we'll
        have at least beam_size non <EOS> extensions if we extract the next top
        2*beam words.
        Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
        https://arxiv.org/abs/1609.08144.

        Args:
        i: loop index
        alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
        alive_log_probs: probabilities of these sequences. [batch_size, beam_size]
        states: dict (possibly nested) of decoding states.
        Returns:
        Tuple of
            (Topk sequences extended by the next word,
            The log probs of these sequences,
            The scores with length penalty of these sequences,
            Flags indicating which of these sequences have finished decoding,
            dict of transformed decoding states)
        """
        # Get the logits for all the possible next symbols    
        flat_ids = torch.reshape(alive_seq, [batch_size * beam_size, -1])
        # (batch_size * beam_size, decoded_length)  
        flat_logits = symbols_to_logits_fn(flat_ids)
        logits = torch.reshape(flat_logits, [batch_size, beam_size, -1])
        # Convert logits to normalized log probs
        candidate_log_probs = torch.log(logits)
        # Multiply the probabilities by the current probabilities of the beam.
        # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
        log_probs = candidate_log_probs + torch.unsqueeze(alive_log_probs, 2)
        length_penalty = torch.pow(((5. + torch.tensor(i+1, dtype=torch.float)) / 6.), alpha)
        curr_scores = log_probs / length_penalty
        # Flatten out (beam_size, vocab_size) probs in to a list of possibilities
        flat_curr_scores = torch.reshape(curr_scores, [-1, beam_size * vocab_size])    
        topk_scores, topk_ids = torch.top_k(flat_curr_scores, k=beam_size * 2)#[batch_size,beam_size*2]
        # Recovering the log probs because we will need to send them back
        topk_log_probs = topk_scores * length_penalty
        # Work out what beam the top probs are in.
        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size  # Unflatten the ids  
        # The next three steps are to create coordinates for tf.gather_nd to pull
        # out the correct sequences from id's that we need to grow.
        # We will also use the coordinates to gather the booleans of the beam
        # items that survived.
        batch_pos = compute_batch_indices(batch_size, beam_size * 2)
        # top beams will give us the actual coordinates to do the gather.
        # stacking will create a tensor of dimension batch * beam * 2, where the
        # last dimension contains the i,j gathering coordinates.
        topk_coordinates = torch.stack([batch_pos, topk_beam_index], axis=2)
        seq_indexes = compute_seq_length(topk_beam_index, alive_seq.shape[-1])
        # Gather up the most probable 2*beams both for the ids and
        # finished_in_alive bools
        topk_seq = torch.gather(alive_seq,1, seq_indexes)        
        # Append the most probable alive
        topk_seq = torch.concat([topk_seq, torch.unsqueeze(topk_ids, 2)], axis=2)   
        topk_finished = (topk_ids== eos_id)
        return topk_seq, topk_log_probs, topk_scores, topk_finished, states   
    
    def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                 finished_flags, states):
        """Inner beam search loop.

        There are three groups of tensors, alive, finished, and topk.
        The alive group contains information about the current alive sequences
        The topk group contains information about alive + topk current decoded words
        the finished group contains information about finished sentences, that is,
        the ones that have decoded to <EOS>. These are what we return.
        The general beam search algorithm is as follows:
        While we haven't terminated (pls look at termination condition)
        1. Grow the current alive to get beam*2 topk sequences
        2. Among the topk, keep the top beam_size ones that haven't reached EOS
        into alive
        3. Among the topk, keep the top beam_size ones have reached EOS into
        finished
        Repeat
        To make things simple with using fixed size tensors, we will end
        up inserting unfinished sequences into finished in the beginning. To stop
        that we add -ve INF to the score of the unfinished sequence so that when a
        true finished sequence does appear, it will have a higher score than all the
        unfinished ones.

        Args:
        i: loop index
        alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
        alive_log_probs: probabilities of the beams. [batch_size, beam_size]
        finished_seq: Current finished sequences.
            [batch_size, beam_size, i+1]
        finished_scores: scores for each of these sequences.
            [batch_size, beam_size]
        finished_flags: finished bools for each of these sequences.
            [batch_size, beam_size]
        states: dict (possibly nested) of decoding states.

        Returns:
        Tuple of
            (Incremented loop index
            New alive sequences,
            Log probs of the alive sequences,
            New finished sequences,
            Scores of the new finished sequences,
            Flags indicating which sequence in finished as reached EOS,
            dict of final decoding states)
        """

        # Each inner loop, we carry out three steps:
        # 1. Get the current topk items.
        # 2. Extract the ones that have finished and haven't finished
        # 3. Recompute the contents of finished based on scores.
        topk_seq, topk_log_probs, topk_scores, topk_finished = grow_topk(
            i, alive_seq, alive_log_probs)
        alive_seq, alive_log_probs, _ = grow_alive(
            topk_seq, topk_scores, topk_log_probs, topk_finished)
        finished_seq, finished_scores, finished_flags, _ = grow_finished(
            finished_seq, finished_scores, finished_flags, topk_seq, topk_scores,
            topk_finished)

        return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
                finished_flags)   
    
    def _is_not_finished(i,  alive_log_probs,finished_scores):
        """Checking termination condition.

        We terminate when we decoded up to decode_length or the lowest scoring item
        in finished has a greater score that the highest prob item in alive divided
        by the max length penalty

        Args:
        i: loop index
        alive_log_probs: probabilities of the beams. [batch_size, beam_size]
        finished_scores: scores for each of these sequences.
            [batch_size, beam_size]

        Returns:
        Bool.
        """
        max_length_penalty = torch.pow(((5. + torch.tensor(decode_length, dtype=torch.float)) / 6.), alpha)
        # The best possible score of the most likely alive sequence.
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

        if not stop_early:
        # by considering the min score (in the top N beams) we ensure that
        # the decoder will keep decoding until there is at least one beam
        # (in the top N) that can be improved (w.r.t. the alive beams).
        # any unfinished beam will have score -INF - thus the min
        # will always be -INF if there is at least one unfinished beam -
        # which means the bound_is_met condition cannot be true in this case.
            lowest_score_of_finished_in_finished,_ = torch.min(finished_scores, axis=1)
        else:
        # by taking the max score we only care about the first beam;
        # as soon as this first beam cannot be beaten from the alive beams
        # the beam decoder can stop.
        # similarly to the above, if the top beam is not completed, its
        # finished_score is -INF, thus it will not activate the
        # bound_is_met condition. (i.e., decoder will keep going on).
        # note we need to find the max for every sequence eparately - so, we need
        # to keep the batch dimension (see axis=1)
            lowest_score_of_finished_in_finished,_ = torch.max(finished_scores,axis=1)
        bound_is_met = torch.all(torch.gt(lowest_score_of_finished_in_finished,
                   lower_bound_alive_scores))

        return torch.logical_and(
            torch.lt(i, decode_length), torch.logical_not(bound_is_met))
    
    cur_len = torch.tensor(0, dtype=torch.int)
    while(_is_not_finished(cur_len, alive_log_probs, finished_scores)):
        cur_len, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags=\
            inner_loop(cur_len, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags)
    # Accounting for corner case: It's possible that no sequence in alive for a
    # particular batch item ever reached EOS. In that case, we should just copy
    # the contents of alive for that batch item. tf.reduce_any(finished_flags, 1)
    # if 0, means that no sequence for that batch index had reached EOS. We need
    # to do the same for the scores as well.
    alive_seq = alive_seq.reshape((batch_size, beam_size, -1))
    finished_seq=finished_seq.reshape((batch_size, beam_size, -1))
    all_finished = torch.any(finished_flags, 1).reshape(batch_size,1,1)
    all_finished = all_finished.repeat(1,beam_size,alive_seq.shape[-1])
    finished_seq = torch.where(all_finished, finished_seq, alive_seq)
    finished_scores = torch.where(all_finished, finished_scores, alive_log_probs)
    return finished_seq, finished_scores, states
