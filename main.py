from Grid import Grid


def main():
    grid = Grid(-2, 2, -2, 2, (1080, 1080))
    grid.newton_iter()
    grid.gen_image()


if __name__ == "__main__":
    main()
