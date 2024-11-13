from agents.brain import Brain, BrainValue
from environment.game import Game
from utils.brain_trainer import BrainTrainer
from utils.config import Config


def main():
    config = Config("config.json")
    game = Game(config)
    game.run()


def train():
    brain = Brain(456, 128)
    critic = BrainValue(456, 128)
    # brain.load("weights/0105.pth")
    config = Config("config.json")
    trainer = BrainTrainer(brain, config, critic)
    for epoch in range(0, 9996):
        print(f"Starting epoch {epoch}")
        trainer.train_episode()
        trainer.decay_epsilon()
        if epoch % 5 == 0 and config["learn"]:
            brain.save(f"weights/{epoch:0>4}.pth")


if __name__ == "__main__":
    # main()
    train()
