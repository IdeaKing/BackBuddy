from gooey import Gooey, GooeyParser
import cv2

@Gooey(    
    dump_build_config = False,
    auto_start=False,
    default_size=(1000, 1000),
    program_name = "BackBuddies Trainer Analysis Program",
    image_dir="docs") # BackBuddiesLogo.png")
def gui_args():
    parser = GooeyParser(
        description = "Your Personal Trainer")
    parser_group = parser.add_argument_group(
        "Select files")
    parser_group.add_argument(
        "--Video-File",
         "-v",
         help = "Path to the input file with video.", 
         widget = "FileChooser")
    """
    parser_group = parser.add_argument_group(
        "Select Camera if Needed")
    parser_group.add_argument(
        "--Camera-On",
        "-o",
        help = "Select Camera (for live feed)",
        widget = "BlockCheckbox")
    parser_group.add_argument(
        "--Camera-Number",
        "-n",
        help = "Select Camera (for live feed)",
        widget = "IntegerField")
    """
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    gui_args()