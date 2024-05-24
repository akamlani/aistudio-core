from pydantic import BaseModel

def trsfrm_pydantic_to_dict(pydantic_obj:BaseModel) -> dict:
    """tranform a pyndantic object to a dictionary

    Args:
        pydantic_obj (BaseModel): instantiated pydantic object

    Returns:
        dict: dictionary representation of the pydantic object

    Example:
    from pydantic import BaseModel
    class User (BaseModel):
        name: str
        age:  int
        location: str

    >>> trsfrm_pydantic_to_dict ( User(name='john doe', age=47, location='NYC') )
    """
    return pydantic_obj.model_dump()

def trsfrm_pydantic_to_schema(pydantic_cls:BaseModel) -> dict:
    """generate a json schema from a pydantic class

    Args:
        pydantic_cls (BaseModel): name of class

    Returns:
        dict: schema representation

    Example:
    from pydantic import BaseModel
    class User (BaseModel):
        name: str
        age:  int
        location: str

    >>> trsfrm_pydantic_to_schema ( User )
    """
    return pydantic_cls.model_json_schema()