from constants import *
from grid import Grid
from gui import Gui

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="A-star Algorithm")
    parser.add_argument("--m", type=int, default=30, help="Number of rows")
    parser.add_argument("--n", type=int, default=30, help="Number of columns")
    parser.add_argument("--obs", type=float, default=0.2, help="Obstacle ratio")
    args = parser.parse_args()

    grid = Grid(args.m, args.n, args.obs)
    gui = Gui(grid)
    gui.run()

if __name__ == "__main__":
    main()
    