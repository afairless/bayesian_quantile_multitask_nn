#! /usr/bin/env python3

from src.s01_generate_data import (
    create_data_with_parameters, 
    split_data_with_parameters)


def main():

    mvn_components = create_data_with_parameters()
    data = split_data_with_parameters(mvn_components.cases_data)


if __name__ == '__main__':
    main()
