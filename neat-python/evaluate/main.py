import visualize, neat, gym, sys

if len(sys.argv) < 2:
    print("Usage: {} path-to-dir".format(sys.argv[0]))
    exit()

dire = sys.argv[1]
print("Dir: {}".format(dire))

dire = './' + dire + '/'

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, dire + "fc.config")
winner = None

def get_winner(p):
    max_fitness = -9999999999
    best_genome = None
    for v in p.population:
        genome = p.population[v]
        if genome.fitness > max_fitness:
            max_fitness = genome.fitness
            winner = genome
    print(max_fitness)
    return([max_fitness, winner])

mf = -999999999

for x in range(100):
    try:
        fn = dire + 'neat-checkpoint-' + str(x)
        p = neat.Checkpointer.restore_checkpoint(fn)
        print(fn)
        [mf_, winner_] = get_winner(p)
        if mf_ > mf:
            mf = mf_
            winner = winner_
    except:
        pass

print("Max score: {}".format(mf))
visualize.draw_net(config, winner, False)
