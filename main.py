from environment.game import Game
from utils.config import Config


def main():
    config = Config("config.json")
    game = Game(config)
    game.run()


if __name__ == "__main__":
    main()
