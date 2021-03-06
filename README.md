# hse-ml-practices

**data** — папка для хранения данных

*raw* — папка для исходных (сырых) данных

*interim* — папка для промежуточных, преобразованных данных

*processed* — папка для данных, подготовленных для обучения моделей (финальные датасеты)

*external* — папка для данных из сторонних источников, либо дополнительное обогащение исходных данных, либо дополнительные данные, полученные в процессе исследования

**docs** — папка для документации

**reports** — папка для сгенерированных отчетов, содержащих результаты анализа в форматах HTML, PDF, LaTeX и т. д.

*figures* — папка для сгенерированных графиков и картинок для использования в отчетности

**models** — папка для хранения обученных и сериализованных моделей, весов моделей или любой другой способ для сохранения результатов обучения

**src** — папка для исходного кода, также в этой папке размещается __init__.py файл для использования исходного кода из этой папки как python модуль

*data* — папка для скриптов для загрузки, генерации данных или преобразования данных

*features* — папка для скриптов для преобразования данных в признаки для обучения и использования модели

*models* — папка для скриптов для обучения моделей и использования обученных моделей для прогнозирования

*reports* — папка для скриптов для создания исследовательских отчетов и визуализаций (графиков и картинок)

**references** — папка для хранения пояснительных материалов, руководства, словари с терминами и другая полезная информация для понимая задачи и данных

**notebooks** — Jupyter notebooks. Либо другие инструменты для интерактивного исследования, которые полезны на ранних этапах исследований

**workflow** — папка для хранения вспомогательных файлов систем управления рабочим процессом (airflow, snakemake, Cmake и т.д.)

*envs* — папка для хранения файлов виртуального окружения для исполнения шагов пайплайна

*rules* — папка для отдельных шагов пайплайна, если он будет большим, легче разбить на отдельные файлы для удобства

*scripts* — папка для вспомогательных скриптов для выполнения пайплайна

*config* — файл для описания конфигурации для пайплайна

*build file* — файл системы управления рабочим процессом, описывающий пайплайны исследований

**readme** — readme верхнего уровня для разработчиков

**.env** — файл для хранения ключей к базам, переменных среды и подобной конфиденциальной информации

**requirements** — файл для описания рабочего окружения исследователей, файл с пакетами виртуального окружения, которое использовали разработчики
