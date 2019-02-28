def pagerank(G, beta=0.85, iteration_count=100, teleport_lst=None, eps=1e-8):
    
    if not teleport_lst:
        teleport_lst = G.keys()

    N = len(G.keys())
    next_rank_lst = [1/N for _ in range(N)]
    current_rank_lst = next_rank_lst[:]
    
    teleport_lst_count = len(teleport_lst)
    
    for i in range(iteration_count):
        current_rank_lst, next_rank_lst = next_rank_lst, current_rank_lst
        for j in range(N):
            next_rank_lst[j] = 0
        for node in teleport_lst:
            next_rank_lst[node] = (1 - beta) / teleport_lst_count
        for node in G:
            if G[node]:
                contribution = beta * (current_rank_lst[node] / len(G[node]))
                for edge in G[node]:
                    next_rank_lst[edge] += contribution
        
        leakage_contribution = (1 - sum(next_rank_lst)) / N
        for j in range(N):
            next_rank_lst[j] += leakage_contribution
        
        total_diff = 0
        for c, n in zip(current_rank_lst, next_rank_lst):
            total_diff += abs(c - n)
        
        if total_diff < eps:
            return next_rank_lst
    
    return next_rank_lst