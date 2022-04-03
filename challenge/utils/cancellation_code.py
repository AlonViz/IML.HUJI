import re


def process_cancellation_code(code):
    regex_days_before = "^(([0-9]+)D)(([0-9]+)N|([0-9]+)P)"
    regex_no_show = "(([0-9]+)P|([0-9]+)N)"
    options = re.split("_", code)
    final = []
    for option in options:
        days_match = re.match(regex_days_before, option)
        if days_match:
            days_before = None if days_match.group(2) is None else int(days_match.group(2))
            nights_to_pay = None if days_match.group(4) is None else int(days_match.group(4))
            percentage = None if days_match.group(5) is None else int(days_match.group(5))
            final.append([days_before, nights_to_pay, percentage])
            continue
        no_show_match = re.match(regex_no_show, option)
        if no_show_match:
            nights_to_pay = None if no_show_match.group(3) is None else int(no_show_match.group(3))
            percentage = None if no_show_match.group(2) is None else int(no_show_match.group(2))
            final.append([0, nights_to_pay, percentage])

    return final


def evaluate_cancellation_code(cancellation_code: str, booking_time_before: int, stay_duration: int) -> float:
    """
    gives a numerical value to given cancellation code, return expected fine in percentage
    :return:
    """
    cancellations = process_cancellation_code(cancellation_code)
    p = min(7, booking_time_before)
    chosen_p = min([lst for lst in cancellations if lst[0] > p], key=lambda tup: tup[0], default=[None, None, None])
    expected_fine = 0 if chosen_p[0] is None else chosen_p[2] if chosen_p[1] is None else chosen_p[1] / stay_duration
    return expected_fine


def no_show(cancellation_code: str) -> int:
    """
    returns 1 if the cancellation code contains a no-show fee, and 0 otherwise
    """
    cancellations = process_cancellation_code(cancellation_code)
    return any(lst for lst in cancellations if lst[0] == 0)
