import pytest

from autoslo.utils.class_with_factory import ClassWithFactory


class TestBase(ClassWithFactory):
    # Each subclass of ClassWithFactory should have its own _registry variable,
    # where it maps strings to the types of *its* subclasses.

    @property
    def name(self) -> str:
        """
        Get the name of the ClassWithFactory instance. Should reflect both the
        type of the bottom-level subclass and any key configuration parameters.

        Returns:
            The name of the ClassWithFactory instance.
        """
        # Minimal name implementation for tests
        return self.__class__.__name__


class Foo(TestBase):
    def __init__(self) -> None:
        self.created = True


class Bar(TestBase):
    def __init__(self, a: int, b: str = "x") -> None:
        self.a = a
        self.b = b


class Baz(TestBase):
    def __init__(self, s: str) -> None:
        self.s = s


class Broken(TestBase):
    def __init__(self) -> None:
        raise RuntimeError("boom")


# Register subclasses for the TestBase factory lookup.
TestBase._registry = {"Foo": Foo, "Bar": Bar, "Baz": Baz, "Broken": Broken}


def test_from_name_basic_instantiation() -> None:
    """
    Basic usage: instantiate a registered subclass with no constructor

    args.
    """
    inst = TestBase.from_name("Foo")
    assert isinstance(inst, Foo)
    assert getattr(inst, "created", False) is True


def test_from_name_with_literal_kwargs() -> None:
    """
    Parse and apply literal keyword arguments via ast.literal_eval to
    initialize the subclass.
    """
    inst = TestBase.from_name("Bar(a=5,b='hello')")
    assert isinstance(inst, Bar)
    assert inst.a == 5
    assert inst.b == "hello"


def test_from_name_unquoted_string_fallback() -> None:
    """
    If ast.literal_eval fails for a value, the raw unquoted string should
    be used as the fallback value.
    """
    inst = TestBase.from_name("Baz(s=unquoted_value)")
    assert isinstance(inst, Baz)
    assert inst.s == "unquoted_value"


def test_from_name_unknown_subclass_raises() -> None:
    """
    Requesting an unknown subclass name should raise a ValueError with a
    helpful message.
    """
    with pytest.raises(ValueError):
        TestBase.from_name("NoSuchClass")


def test_from_name_malformed_string_raises() -> None:
    """
    If the inside of the parentheses contains an entry without '=' the
    function should raise a ValueError about a malformed string.
    """
    with pytest.raises(ValueError):
        TestBase.from_name("Foo(badentry)")


def test_from_name_instantiation_error_wrapped() -> None:
    """
    Exceptions raised during subclass instantiation should be wrapped as a
    ValueError by from_name.
    """
    with pytest.raises(ValueError) as excinfo:
        TestBase.from_name("Broken")
    assert "Error instantiating" in str(excinfo.value)
