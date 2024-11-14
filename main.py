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
    load_batch = 0
    if load_batch > 0:
        brain.load(f"weights/b{load_batch:0>5}.pth")
        critic.load(f"weights/c{load_batch:0>5}.pth")
    config = Config("config.json")
    trainer = BrainTrainer(brain, config, critic)
    for epoch in range(load_batch + 1, 3001):
        print(f"Starting epoch {epoch}")
        trainer.train_episode()
        trainer.decay_epsilon()
        if epoch % 5 == 0 and config["learn"]:
            brain.save(f"weights/b{epoch:0>5}.pth")
            critic.save(f"weights/c{epoch:0>5}.pth")


if __name__ == "__main__":
    # main()
    train()
