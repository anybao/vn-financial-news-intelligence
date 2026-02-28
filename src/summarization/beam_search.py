import torch
from queue import PriorityQueue

class BeamSearchNode:
    """Helper class for holding states during beam search."""
    def __init__(self, hiddenstate, cellstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.c = cellstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=0.7):
        """Length penalized evaluation score."""
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) ** alpha
        
    def __lt__(self, other):
        return self.eval() < other.eval()

def decode_beam_search(decoder, encoder_outputs, encoder_hidden, encoder_cell, sos_idx, eos_idx, max_len, beam_width, device):
    """
    Performs beam search decoding for a single batch element.
    Normally, this would be optimized for batched processing, but loop-based is standard for educational baselines.
    """
    # Start node
    decoder_input = torch.tensor([[sos_idx]], device=device)
    h, c = decoder.init_hidden(encoder_hidden, encoder_cell)
    
    node = BeamSearchNode(h, c, None, sos_idx, 0, 1)
    
    # Queue for holding open nodes
    nodes = PriorityQueue()
    nodes.put((-node.eval(), node))
    qsize = 1
    
    end_nodes = []
    
    while qsize > 0:
        score, n = nodes.get()
        
        if n.wordid == eos_idx and n.prevNode != None:
            end_nodes.append((score, n))
            if len(end_nodes) >= beam_width:
                break
            continue
            
        if n.leng >= max_len:
            end_nodes.append((score, n))
            if len(end_nodes) >= beam_width:
                break
            continue
            
        decoder_input = torch.tensor([[n.wordid]], device=device)
        output, h, c, _ = decoder(decoder_input, n.h, n.c, encoder_outputs)
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        
        # Get topk
        topk_log_probs, topk_word_ids = torch.topk(log_probs, beam_width)
        
        for k in range(beam_width):
            next_word = topk_word_ids[0][k].item()
            next_log_prob = n.logp + topk_log_probs[0][k].item()
            next_node = BeamSearchNode(h, c, n, next_word, next_log_prob, n.leng + 1)
            nodes.put((-next_node.eval(), next_node))
            
        qsize += beam_width - 1
        
    if len(end_nodes) == 0:
        end_nodes = [nodes.get() for _ in range(beam_width)]
        
    # Get the best node
    _, best_node = sorted(end_nodes, key=lambda x: x[0])[0] # The score in priority queue is negative, so smallest is best or we sort reverse
    
    # Backtrack to get words
    path = []
    while best_node.prevNode is not None:
        path.append(best_node.wordid)
        best_node = best_node.prevNode
        
    return path[::-1] # Reverse the path
