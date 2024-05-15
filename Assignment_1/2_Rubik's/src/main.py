from cube import Cube
from gui import Gui


def main():
    cube = Cube(2)
    gui = Gui(cube)
    gui.run()

if __name__ == "__main__":
    main()
        