from agents.brain import Brain, BrainValue
from environment.game import Game
from utils.brain_trainer import BrainTrainer
from utils.config import Config


def main():
    config = Config("config.json")
    game = Game(config)
    game.run()


def train():
    brain = Brain(166, 256)
    critic = BrainValue(166, 256)

    config = Config("config.json")

    load_batch = 0

    # if not config["learn"]:
    #     brain.eval()

    if load_batch > 0:
        brain.load(f"weights/b{load_batch:0>5}.pth")
        critic.load(f"weights/c{load_batch:0>5}.pth")

    trainer = BrainTrainer(brain, config, critic)
    trainer.epoch = load_batch + 1

    for epoch in range(load_batch + 1, 300):
        print(f"Starting epoch {epoch}")
        trainer.train_episode()
        trainer.decay_epsilon()
        if epoch % 2 == 0 and config["learn"]:
            brain.save(f"weights/b{epoch:0>5}.pth")
            critic.save(f"weights/c{epoch:0>5}.pth")


if __name__ == "__main__":
    # main()
    train()
