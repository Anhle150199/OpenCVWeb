from django import forms
from .models import CV, OjDt

class CVForm(forms.ModelForm):
    class Meta:
        model = CV
        fields = ['image']


class OjDtForm(forms.Form):
    class Meta:
        model = OjDt
        fields = ['image', 'imageTemplate']
GEEKS_CHOICES =(
    ("1", "One"),
    ("2", "Two"),
    ("3", "Three"),
    ("4", "Four"),
    ("5", "Five"),
)
class objCtForm(forms.Form):
    image = forms.ImageField()
    noise = forms.ChoiceField(choices = GEEKS_CHOICES)

