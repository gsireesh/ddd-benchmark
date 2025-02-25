def convert_temperature(value: int, from_unit: str, to_unit: str) -> float:
    """
    Converts temperature values between units, with units represented as c, f, or k
    
    Arguments:
    value (int): The original temperature value, in 'from_unit'
    from_unit (str): the original unit, either c f or k
    to_unit (str): the target unit, either c f or k

    Returns:
    integer representation of temperature.

    Throws:
    ValueError when the unit is not specified correctly.
    """
    from_unit, to_unit = from_unit.upper(), to_unit.upper()
    
    if from_unit == to_unit:
        return float(value)
    
    # Convert from original unit to Celsius
    if from_unit == 'C':
        temp_c = value
    elif from_unit == 'F':
        temp_c = (value - 32) * 5/9
    elif from_unit == 'K':
        temp_c = value - 273.15
    else:
        raise ValueError("Invalid original unit. Use 'C', 'F', or 'K'.")
    
    # Convert from Celsius to desired unit
    if to_unit == 'C':
        return temp_c
    elif to_unit == 'F':
        return temp_c * 9/5 + 32
    elif to_unit == 'K':
        return temp_c + 273.15
    else:
        raise ValueError("Invalid target unit. Use 'C', 'F', or 'K'.")