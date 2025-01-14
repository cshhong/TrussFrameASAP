import matplotlib.pyplot as plt

def format_scientific(num):
    '''
    Custom function to convert large integers to scientific notation with 1 decimal precision
    '''
    if num == 0:
        return "0.0e+00"
    exponent = len(str(num)) - 1
    base = num / 10**exponent
    return f'{base:.1f}e{exponent:+}'

def get_mcts_size(steps):
    '''
    Calculate the number of nodes within state tree for cantilever env 
    Output a list of (i, total) pairs
    '''
    total = 0
    curr_level = 1
    results = []
    
    for i in range(steps):
        total += curr_level
        next_level = curr_level * (curr_level + 2)
        curr_level = next_level
        # Collect the (i, total) pair
        results.append((i, total))
    
    return results

def plot_mcts_results(results):
    '''
    Plot the (i, total) pairs on a graph
    '''
    steps = [i for i, total in results]
    totals = [total for i, total in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, totals, marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('Total Number of Nodes')
    plt.title('MCTS Tree Growth for Cantilever Env')
    plt.xticks(steps)
    # Annotate each point with its corresponding total value
    for i, total in enumerate(totals):
        plt.text(steps[i]-0.1, totals[i], f'{format_scientific(total)}',color='blue', fontsize=8, ha='right', va='bottom')
    
    plt.yscale('log')  # Use log scale if the numbers grow very large
    plt.grid(True)
    plt.show()

# Get the (i, total) pairs for 10 steps
results = get_mcts_size(10)

# Plot the results
plot_mcts_results(results)
