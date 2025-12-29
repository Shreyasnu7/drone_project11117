class AppContract:
    """
    Defines what the app is allowed to send.
    """

    allowed_fields = [
        "intent_input",
        "manual_control",
        "override",
        "telemetry_request",
        "log_request",
    ]