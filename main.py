from environment.game import Game
from utils.config import Config
from utils.brain_trainer import BrainTrainer
from agents.brain import Brain
from agents.brain import BrainValue


def main():
    config = Config("config.json")
    game = Game(config)
    game.run()


def train():
    brain = Brain(16, 128)
    critic = BrainValue(16, 128)
    config = Config("config.json")
    trainer = BrainTrainer(brain, critic, config)
    for epoch in range(20):
        print(f"Starting epoch {epoch}")
        trainer.train_episode()
        # brain.save(f"weights/{epoch}.pth")


if __name__ == "__main__":
    # main()
    train()
