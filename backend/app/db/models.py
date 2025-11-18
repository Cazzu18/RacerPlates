from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, Integer, Text, BigInteger, Boolean

#Declarative Mapping is the typical way that mappings are constructed in modern SQLAlchemy

class Base(DeclarativeBase): pass #when subclassed, this will apply the declarative mapping process to all subclasses that derive from it

class Meal(Base):
    __tablename__ = "meals"
    id:Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement=True)
    menu_item_id: Mapped[int | None] = mapped_column(BigInteger, index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    ingredients: Mapped[str] = mapped_column(Text, default ="")
    allergens: Mapped[str] = mapped_column(Text, default ="")
    station: Mapped[str] = mapped_column(Text, default ="")
    diet_key: Mapped[str] = mapped_column(Text, default ="")
    calories: Mapped[float | None] = mapped_column(Float, nullable = True)
    fat: Mapped[float | None] = mapped_column(Float, nullable = True)
    cholesterol: Mapped[float | None] = mapped_column(Float, nullable = True)
    sodium: Mapped[float | None] = mapped_column(Float, nullable = True)
    carbohydrates: Mapped[float | None] = mapped_column(Float, nullable = True)
    fiber: Mapped[float | None] = mapped_column(Float, nullable = True)
    sugar: Mapped[float | None] = mapped_column(Float, nullable = True)
    protein: Mapped[float | None] = mapped_column(Float, nullable = True)
    iron: Mapped[float | None] = mapped_column(Float, nullable = True)
    calcium: Mapped[float | None] = mapped_column(Float, nullable = True)
    potassium: Mapped[float | None] = mapped_column(Float, nullable = True)
    meal_time: Mapped[str | None] = mapped_column(String(32), nullable = True)
    is_vegan: Mapped[bool] = mapped_column(Boolean, default = False)
    is_vegetarian: Mapped[bool] = mapped_column(Boolean, default = False)
    is_mindful: Mapped[bool] = mapped_column(Boolean, default = False)
    
    
class Rating(Base):
    __tablename__ = "ratings"
    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement=True)
    meal_id: Mapped[int] = mapped_column(Integer, index=True)
    
    label_3class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stars_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    dietary_pref: Mapped[str | None] = mapped_column(Text, nullable = True)
    satisfaction_factor: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
