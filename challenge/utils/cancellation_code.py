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


def filter(cancellation_code: str, booking_time_before: int, stay_duration: int) -> float:
    cancellations = process_cancellation_code(cancellation_code)
    filtered = [i for i in cancellations if i[0] < booking_time_before]
    prec_only = []
    for i in filtered:
        if i[2] is not None:
            prec_only.append([i[0], i[2]])
        else:
            prec_only.append([i[0], i[1] / stay_duration])


def no_show(cancellation_code: str) -> int:
    """
    returns 1 if the cancellation code contains a no-show fee, and 0 otherwise
    """
    cancellations = process_cancellation_code(cancellation_code)
    return any(lst for lst in cancellations if lst[0] == 0)


def fine_after_x_days(cancellation_code: str, booking_time_before: int, stay_duration: int, days: int):
    """
    returns the expected fine in percentages after 'days' days from reservation.
    """
    time_before_reservation = booking_time_before - days
    if time_before_reservation < 0:
        return 0

    cancellations = process_cancellation_code(cancellation_code)

    # convert cancellation policy to format (Days, Percentage)
    percentage_cancellations = []
    for cancel in cancellations:
        if cancel[1] is None:
            percentage_cancellations.append((cancel[0], cancel[2]))
        else:
            percentage_cancellations.append((cancel[0], cancel[1] / stay_duration))

    if not percentage_cancellations:
        return 0
    # return the fine associated with the smallest number of days larger than time_before_reservation
    fines = [x for x in percentage_cancellations if x[0] > time_before_reservation]
    if not fines:
        return 0
    return min(fines, key=lambda x: x[0])[1]
