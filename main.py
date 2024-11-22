from agents.brain import Brain, BrainValue
from agents.DDPGAgent import DDPGAgent
from environment.game import Game
from utils.brain_trainer import BrainTrainer
from utils.config import Config
from utils.DDPGTrainer import DDPGTrainer


def main():
    config = Config("config.json")
    game = Game(config)
    game.run()


def train_brain():
    brain = Brain(166, 256)
    critic = BrainValue(166, 256)

    config = Config("config.json")

    load_batch = 30

    if not config["learn"]:
        brain.eval()

    if load_batch > 0:
        brain.load(f"weights/b{load_batch:0>5}.pth")
        critic.load(f"weights/c{load_batch:0>5}.pth")

    trainer = BrainTrainer(brain, config, critic)
    trainer.epoch = load_batch + 1

    for epoch in range(load_batch + 1, 35):
        print(f"Starting epoch {epoch}")
        trainer.train_episode()
        trainer.decay_epsilon()
        if epoch % 2 == 0 and config["learn"]:
            brain.save(f"weights/b{epoch:0>5}.pth")
            critic.save(f"weights/c{epoch:0>5}.pth")


def train_ddpg():
    config = Config("config.json")
    agent = DDPGAgent(22, 2, 1, 128, config)
    load_number = 0
    if load_number > 0:
        actor_path = f"weights/a{load_number:0>5}.pth"
        critic_path = f"weights/c{load_number:0>5}.pth"
        agent.load(actor_path, critic_path)
    if not config["learn"]:
        agent.eval()
    else:
        agent.train()
    trainer = DDPGTrainer(agent, config)
    trainer.train(1000, load_number + 1)


if __name__ == "__main__":
    # main()
    # train_brain()
    train_ddpg()
