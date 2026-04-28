import argparse


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the current CARLA spectator transform.")
    parser.add_argument("--carla-host", type=str, default="carla", help="IP address or hostname of the CARLA server (default: 'carla')")
    parser.add_argument("--carla-timeout", type=float, default=30.0, help="Timeout of the CARLA server response in seconds (default: 30.0)")
    return parser.parse_args()


def main() -> None:
    opt = arg_parse()
    import carla

    client = carla.Client(opt.carla_host, 2000)
    client.set_timeout(opt.carla_timeout)
    world = client.get_world()

    spectator = world.get_spectator()
    location = spectator.get_transform().location
    rotation = spectator.get_transform().rotation
    print(f"Location: {location.x:.2f}, {location.y:.2f}, {location.z:.2f},")
    print(f"Rotation: {rotation.pitch:.2f}, {rotation.yaw:.2f}, {rotation.roll:.2f}")


if __name__ == "__main__":
    main()
