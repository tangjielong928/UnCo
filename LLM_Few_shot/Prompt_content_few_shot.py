class MLLM_base_generator():
    def __init__(self, data_name = "GMNER"):
        self.data_name = data_name

    def get_system_prompt(self):
        system_prompt = self.Role_Definition() + self.Entity_Type_Description()  + self.Format_output_Description() + self.Action_Definition()
        return system_prompt

    def Role_Definition(self):

        role_describe =  "[Role]\n " \
                        "You are an AI assistant focused on extracting the textual named entity from the provided image.\n" \
                        
        return role_describe

    def Action_Definition(self):

        action_describe = " [Action Required]\n" \
                        " - Carefully review the provided image and the original text. \n"\
                        " - Please think about the following question.:\n " \
                        "1. What entities are there in the original text? \n"\
                        "2. What are the types of these entities? \n"\
                        "3. What are the region-boxes of these entities?\n"\
                        " - Important note: If the entity cannot be precisely located at a specific position within the image, or if the entity encompasses the entire image area, set the bounding box to '[]'. Only output one nearest entity type.\n"\
                        " - Please output your final results according to [Format_output_Description].\n" \
        
        return action_describe

    def Entity_Type_Description(self):

        if 'GMNER' in self.data_name:
            label_describe = "[Entity Type Description] Here are the entity type you need to extract: \n" \
            " - PER (Person): This category includes names of persons, such as individual people or groups of people with personal names.\n "  \
            " - ORG (Organization): The organization category consists of names of companies, institutions, or any other group or entity formed for a specific purpose.\n" \
            " - LOC (Location): The location category represents names of geographical places or landmarks, such as cities, countries, rivers, or mountains. \n" \
            " - MISC (Miscellaneous): The miscellaneous category encompasses entities that do not fall into the above three categories. This includes song or film name, like Ay Ziggy Zoomba, and events, like 1000 Lakes Rally, making it a very diverse category.\n"
        elif 'FMNERG' in self.data_name:
            label_describe = "[Entity Type Description] Here are the entity type (fine-grained) you need to extract: \n" \
            "- city\n- country\n- state\n- continent\n- location_other\n- park\n- road\n- building_other\n- cultural_place\n- entertainment_place\n- sports_facility\n- company\n- educational_institution\n- band\n- government_agency\n- news_agency\n- organization_other\n- political_party\n" \
            "- social_organization\n- sports_league\n- sports_team\n- politician\n- musician\n- actor\n- artist\n- athlete\n- author\n- businessman\n- character\n- coach\n- director\n- intellectual\n- journalist\n- person_other\n- animal\n- award\n- medical_thing\n- website\n- ordinance\n" \
            "- art_other\n- film_and_television_works\n- magazine\n- music\n- written_work\n- event_other\n- festival\n- sports_event\n- brand_name_products\n- game\n- product_other\n- software\n"
            
        else:
            raise Exception("Invalid dataname! Please check")

        return label_describe

    def Format_output_Description(self):

        format_describe = "[Format_output_Description]:\n" \
                        "eg: ```json\n{\n  \"pre_entities\": [\n  {\n  \"phrase\": \"James\",\n  \"entity_type\": \"PER\",\n  \"region_box\": [293, 21, 593, 449]\n  }\n  ]\n}```\n"
        
        return format_describe
