"""Fixture: ROOT-CLI — an `if __name__ == "__main__":` guard that calls a
bare-name function directly."""


def main():
    return 0


if __name__ == "__main__":
    main()
