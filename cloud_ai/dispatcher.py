class OrchestrationDispatcher:
    """
    Dispatches plans to downstream systems.
    """

    def dispatch_to_laptop(
        self,
        plan: dict,
    ):
        """
        Sends plan to Laptop AI (Camera Brain).
        """
        # REST / WebSocket / gRPC
        pass

    def dispatch_to_radxa(
        self,
        safety_constraints: dict,
    ):
        """
        Sends safety envelope to Radxa.
        """
        pass