Нейронная сеть, которую можно обучить распознавать символы =) и (=.
Запуск:
```
python SYMNN.py [-p] [-t]
```
```
[-p] - поиграть с готовой моделью. Рисуешь символ - она угадывает!
[-t] - тренировать модель.
```

Файлы:<br>
```
1. SYMNN.py - сама нейронная сеть.
2. draw_box.py - класс окошка для рисования.
3. data_work.py - набор фукнций для размножения и обработки данных.
4. samples - директория с ручным датасетом и аугментированным
```
Запускать файл `data_work.py` не стоит. Это файл использовался для генерации вариаций изображений.<br>
Если случайно запустить его, не очистив папки `sset_full`, `fset_full`, то получится файловая бомба, так как на каждый пример он создаст сотни новых.
