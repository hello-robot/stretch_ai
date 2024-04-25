import stretch
from pynput import keyboard

keys = {'up': False, 'down': False, 'left': False, 'right': False, 'escape': False}

def on_press(key):
    if key == keyboard.Key.up:
        keys['up'] = True
    elif key == keyboard.Key.down:
        keys['down'] = True
    elif key == keyboard.Key.left:
        keys['left'] = True
    elif key == keyboard.Key.right:
        keys['right'] = True


def on_release(key):
    if key == keyboard.Key.up:
        keys['up'] = False
    elif key == keyboard.Key.down:
        keys['down'] = False
    elif key == keyboard.Key.left:
        keys['left'] = False
    elif key == keyboard.Key.right:
        keys['right'] = False
    elif key == keyboard.Key.esc:
        keys['escape'] = True


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release,
    suppress=True
)

def base_keyboard_teleop(apply_translation_vel=0.12, apply_rotational_vel=0.7):
    print("Use four arrow keys to teleoperate the mobile base around")
    print("Press the 'escape' key to exit")
    while not keys['escape']:
        translation_vel = apply_translation_vel * int(keys['up']) + -1.0 * apply_translation_vel * int(keys['down'])
        rotational_vel = apply_rotational_vel * int(keys['left']) + -1.0 * apply_rotational_vel * int(keys['right'])
        if translation_vel == 0.0 and rotational_vel == 0.0:
            continue
        stretch.set_base_velocity(translation_vel, rotational_vel)
        

if __name__ == "__main__":
    stretch.connect()
    listener.start()
    base_keyboard_teleop()
    listener.stop()
