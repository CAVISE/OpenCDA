import argparse


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set the CARLA spectator location from stdin.")
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

    x, y, z = map(float, input().split(","))
    location = carla.Location(x=x, y=y, z=z)
    rotation = carla.Rotation(pitch=0, yaw=-180, roll=0)
    spectator.set_transform(carla.Transform(location, rotation))


if __name__ == "__main__":
    main()
