from agents.brain import Brain
from environment.game import Game
from utils.brain_trainer import BrainTrainer
from utils.config import Config


def main():
    config = Config("config.json")
    game = Game(config)
    game.run()


def train():
    brain = Brain(16, 64)
    config = Config("config.json")
    trainer = BrainTrainer(brain, config)
    for epoch in range(1000):
        print(f"Starting epoch {epoch}")
        trainer.train_episode()

        brain.save(f"weights/{epoch}.pth")


if __name__ == "__main__":
    # main()
    train()
