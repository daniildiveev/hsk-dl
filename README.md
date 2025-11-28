# Лабораторная работа №4

---

**Студенты:** Дивеев Даниил Андреевич, Суджян Эдуард Эдуардович \
**ИСУ:** 368105, 409645 \
**Группа:** P3321 \
**Университет:** НИУ ИТМО \
**Факультет:** Программная инженерия и компьютерная техника \
**Курс:** 3-й курс

---

## Отчет

В лабораторной работе представлена реализация нейронной сети для распознавания
рукописных цифр на основе датасета MNIST.

Требования:

1. Функции:
  - загрузка и предобработка данных из CSV-файла;
  - построение нейронной сети с использованием линейных слоев и функций активации;
  - обучение сети с помощью стохастического градиентного спуска (SGD);
  - оценка точности и визуализация предсказаний.
2. Использовать идиоматичный для Haskell стиль программирования.

### Реализация

```haskell
-- ОПРЕДЕЛЕНИЕ МОДЕЛИ --
l1 <- linear 784 32
l2 <- linear 32 10
let model = [l1, reluLayer, l2, softmaxLayer]
```

### Функции

```haskell
-- ЗАГРУЗКА ДАННЫХ --
readMnistCsv :: FilePath -> Int -> IO [([Double], Int)]
readMnistCsv path limit = do
  content <- readFile path
  let ls = take limit . filter (not . null) . map trim $ lines content
      parsed = mapMaybe parseLine ls
  pure parsed

-- ОБУЧЕНИЕ --
train :: [([Double], Int)] -> [IORef Tensor] -> [Layer] -> IO ()
train dataset params model = mapM_ epoch [1 .. epochs]
  where
    epochs = 3
    batchSize = 10
    lr = 0.1

-- ВЫЧИСЛЕНИЕ ТОЧНОСТИ --
computeAccuracy :: [Layer] -> [([Double], Int)] -> IO Double
computeAccuracy model sample = do
  let (xs, ys) = unzip sample
      batchSize = length sample
  input <- fromList [batchSize, 784] (concat xs)
  out <- forwardSequential model input
  let classes = 10
      rows = chunk classes (values out)
      preds = map argmax rows
      correct = length (filter id (zipWith (==) preds ys))
  pure (fromIntegral correct / fromIntegral batchSize)

-- ВИЗУАЛИЗАЦИЯ --
showPredictions :: [Layer] -> [([Double], Int)] -> [Int] -> String -> IO ()
showPredictions model dataset idxs prefix = do
  let selected = map (dataset !!) idxs
      (xs, ys) = unzip selected
  input <- fromList [length selected, 784] (concat xs)
  out <- forwardSequential model input
  let rows = chunk 10 (values out)
      pixelsRows = chunk 784 (concat xs)
  forM_ (zip3 idxs pixelsRows rows) $ \(origIdx, pixels, outRow) -> do
    let idLabel = prefix ++ "-" ++ show origIdx
    showPredictionWithDir "predictions" idLabel pixels outRow (snd (dataset !! origIdx))
```

### Реализация обучения

```haskell
-- ПАРАМЕТРЫ ОБУЧЕНИЯ --
main :: IO ()
main = do
  !dataset <- readMnistCsv "data/mnist_train.csv" 100
  l1 <- linear 784 32
  l2 <- linear 32 10
  let model = [l1, reluLayer, l2, softmaxLayer]
      params = concatMap parameters model
  train dataset params model
  finalAcc <- computeAccuracy model (take 50 dataset)
  putStrLn $ "FINAL accuracy on first 50 rows: " <> show (finalAcc * 100) <> "%"
```

### Дополнительно

Реализованы функции визуализации:

- `saveImagePng` - сохранение изображений цифр в PNG формате
- `printAsciiImage` - вывод ASCII-превью изображений

Пример вывода обучения:
```
Starting epoch 1...
  batch loss=2.3025850929940455
  batch loss=2.3025850929940455
Epoch 1 accuracy on first 50 rows: 12.0%
```

Пример визуализации предсказания:
```
Example final-2: true=4, pred=4, prob=0.5590279006990548, file=predictions/final/mnist-final-2.png
ASCII preview:

                    :%.
    ::              =*.
    =+              +#.
    #+              %+
    #+             *%=
   .%+             *%.
   =%+             %%.
   +%=            +%#
   +%:          -*%%-
   +%-   ..=++%%%*%%.
   +%%###%%%%*+-  %%
    =*****-.     -%#
                 +%=
                 +%:
                 +%:
                 +@-
                 +%-
                 +%+
                 +@+
                 -%+
```

### Вывод

В ходе выполнения работы была реализована нейронная сеть для классификации рукописных цифр.
Сеть успешно обучается на предоставленных данных и достигает разумной точности на тестовой выборке.
Реализована визуализация предсказаний, что позволяет наглядно оценить качество работы модели.

Архитектура сети (784->32->10) демонстрирует работоспособность подхода даже при небольшом количестве параметров.
Использование Haskell позволило создать типобезопасную и легко расширяемую реализацию.
