class explanation(object):
    """Explanation class to export explainers to JSON.
    
    """

    def __init__(self):
        """Default constructor.

        """
        self.__explanation = []
    
    def add_explanation(self, exp):
        """Add an explanation to the list.

        Args:
            exp (:obj:`alibi.api.interfaces.Explanation`): an explanation to be added.
        """
        self.__explanation.append(exp)

    def to_json(self):
        """converts list to JSON format

        Returns:
            str: a string containing the JSON
        """
        json = ""
        index = 0
        for exp in self.__explanation:
            json += '"' + str(index) + '": '
            json += exp.to_json()
            json += ','
            index+=1
        return json[:-1]

    def save_json(self, filename):
        """Save list to JSON file

        Args:
            filename (:obj:`str`): file location.
        """
        contents = self.to_json()
        file = open(filename, "w")
        file.write(contents)
        file.close()