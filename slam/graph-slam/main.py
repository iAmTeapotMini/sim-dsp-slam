import ex2 as ex

if __name__ == '__main__':
    filename = 'data/simulation-pose-landmark.g2o'
    graph = ex.read_graph_g2o(filename)

    numIterations = 100
    ex.run_graph_slam(graph, numIterations)