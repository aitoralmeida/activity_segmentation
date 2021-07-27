from scipy import spatial

def calculate_context_similarity(actions, embedding_action_matrix, position, window):
    target_action_vector = embedding_action_matrix[actions[position]]
    target_action_vector_context_sim = 0.0
    counter = window * 2
    for i in range(1, window+1):
        # right context we search for similarity
        if position+i < len(actions):
            right_sim = max(0, 1 - spatial.distance.cosine(target_action_vector, embedding_action_matrix[actions[position+i]]))
            target_action_vector_context_sim += right_sim
        else:
            counter -= 1
        # left context we search for disimilarity (1 - similarity)
        if position-i >= 0:
            left_sim = 1 - max(0, 1 - spatial.distance.cosine(target_action_vector, embedding_action_matrix[actions[position-i]]))
            target_action_vector_context_sim += left_sim
        else:
            counter -= 1
    target_action_vector_context_sim = target_action_vector_context_sim / counter
    return target_action_vector_context_sim