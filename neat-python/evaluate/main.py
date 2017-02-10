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
    return(winner)

for x in range(100):
    global winner
    try:
        fn = dire + 'neat-checkpoint-' + str(x)
        p = neat.Checkpointer.restore_checkpoint(fn)
        print(fn)
        winner = get_winner(p)
    except:
        pass

visualize.draw_net(config, winner, False)
