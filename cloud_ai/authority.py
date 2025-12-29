class AuthorityManager:
    """
    Resolves who is allowed to command the drone.
    """

    def resolve(
        self,
        human_override: bool,
        ai_request: bool,
    ) -> str:
        """
        Returns authority owner.
        """

        if human_override:
            return "human"

        if ai_request:
            return "ai"

        return "human"