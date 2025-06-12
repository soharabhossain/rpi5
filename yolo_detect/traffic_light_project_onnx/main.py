from core.controller import TrafficController
from ui.gui_window import TrafficGUI

if __name__ == "__main__":
    controller = TrafficController()
    gui = TrafficGUI(controller)
    gui.run()
