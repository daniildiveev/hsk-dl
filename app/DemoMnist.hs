{-# LANGUAGE BangPatterns #-}
module Main where

import Core.Tensor
import Data.IORef (IORef)
import Data.List (maximumBy, foldl')
import Data.Ord (comparing)
import Data.Char (isSpace)
import Data.Maybe (mapMaybe)
import Text.Read (readMaybe)
import NN.Layer
import NN.Loss
import NN.Optim.SGD

import Codec.Picture
import Control.Monad (forM_, when)
import System.FilePath ((</>), takeDirectory)
import System.Directory (createDirectoryIfMissing)

main :: IO ()
main = do
  putStrLn "Using small network (784->32->10)."
  putStrLn "Loading CSV..."
  !dataset <- readMnistCsv "data/mnist_train.csv" 100
  putStrLn $ "Loaded " <> show (length dataset) <> " rows."
  l1 <- linear 784 32
  l2 <- linear 32 10
  let model = [l1, reluLayer, l2, softmaxLayer]
      params = concatMap parameters model
  train dataset params model
  finalAcc <- computeAccuracy model (take 50 dataset)
  putStrLn $ "FINAL accuracy on first 50 rows: " <> show (finalAcc * 100) <> "%"
  putStrLn "Saving final examples (3-4)..."
  showPredictions model dataset (take 4 [0 ..]) "final"

train :: [([Double], Int)] -> [IORef Tensor] -> [Layer] -> IO ()
train dataset params model = mapM_ epoch [1 .. epochs]
  where
    epochs = 3
    batchSize = 10
    lr = 0.1
    epoch e = do
      putStrLn $ "Starting epoch " <> show e <> "..."
      let batches = chunk batchSize dataset
      mapM_ (trainBatch lr params model) batches
      putStrLn ("Epoch " <> show e <> " done.")
      acc <- computeAccuracy model (take 50 dataset)
      putStrLn $ "Epoch " <> show e <> " accuracy on first 50 rows: " <> show (acc * 100) <> "%"
      let idxs = take 2 [0 ..]
      showPredictions model dataset idxs ("epoch-" ++ show e)

trainBatch :: Double -> [IORef Tensor] -> [Layer] -> [([Double], Int)] -> IO ()
trainBatch lr params model batch = do
  let (xs, ys) = unzip batch
      batchSize = length batch
  input <- fromList [batchSize, 784] (concat xs)
  zeroGradParameters params
  out <- forwardSequential model input
  loss <- nllLoss out ys
  backward loss
  sgdStep lr params
  let [lossVal] = values loss
  putStrLn ("  batch loss=" <> show lossVal)

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

showPredictions :: [Layer] -> [([Double], Int)] -> [Int] -> String -> IO ()
showPredictions model dataset idxs prefix = do
  let n = length dataset
      validIdxs = filter (\i -> i >= 0 && i < n) idxs
  when (null validIdxs) $
    putStrLn $ "No valid indices to show for " <> prefix
  let selected = map (dataset !!) validIdxs
      (xs, ys) = unzip selected
      batchSize = length selected
  input <- fromList [batchSize, 784] (concat xs)
  out <- forwardSequential model input
  let classes = 10
      rows = chunk classes (values out)
      pixelsRows = chunk 784 (concat xs)
  let outDir = "predictions" </> prefix
  createDirectoryIfMissing True outDir
  forM_ (zip3 validIdxs pixelsRows rows) $ \(origIdx, pixels, outRow) -> do
    let idLabel = prefix ++ "-" ++ show origIdx
    showPredictionWithDir outDir idLabel pixels outRow (snd (dataset !! origIdx))

showPredictionWithDir :: FilePath -> String -> [Double] -> [Double] -> Int -> IO ()
showPredictionWithDir outDir idLabel pixels outRow trueLabel = do
  let probs = outRow
      predLabel = argmax outRow
      topProb = maximum probs
      fileName = outDir </> ("mnist-" ++ idLabel ++ ".png")
  createDirectoryIfMissing True outDir
  saveImagePng fileName pixels
  putStrLn $ "Example " <> idLabel <> ": true=" <> show trueLabel <> ", pred=" <> show predLabel <> ", prob=" <> show topProb <> ", file=" <> fileName
  putStrLn "ASCII preview:"
  printAsciiImage pixels
  putStrLn ""

zip4 :: [a] -> [b] -> [c] -> [d] -> [(a,b,c,d)]
zip4 (a:as) (b:bs) (c:cs) (d:ds) = (a,b,c,d) : zip4 as bs cs ds
zip4 _ _ _ _ = []

argmax :: [Double] -> Int
argmax xs = fst (maximumBy (comparing snd) (zip [0 ..] xs))

readMnistCsv :: FilePath -> Int -> IO [([Double], Int)]
readMnistCsv path limit = do
  content <- readFile path
  let !forced = length content
      ls =
        take limit
          . filter (not . null)
          . map trim
          $ lines content
      parsed = mapMaybe parseLine ls
      !_ = foldl' (\acc (xs, y) -> acc + sum xs + fromIntegral y) (0 :: Double) parsed
  if null parsed
    then error "No rows parsed; is the CSV path correct and header removed?"
    else pure parsed
  where
    parseLine line = do
      let cells = splitCommas line
      (labelStr, pixelStrs) <- case cells of
        [] -> Nothing
        (l : rest) -> Just (l, rest)
      label <- readMaybe labelStr
      pixels <- mapM readMaybe pixelStrs
      let xs = map (\p -> realToFrac p / 255) pixels
      if length xs < 784
        then Nothing
        else Just (take 784 xs, label)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

splitCommas :: String -> [String]
splitCommas [] = [""]
splitCommas (',' : xs) = "" : splitCommas xs
splitCommas (c : xs) =
  case splitCommas xs of
    [] -> [[c]]
    w : rest -> (c : w) : rest

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs =
  let (h, rest) = splitAt n xs
   in h : chunk n rest

saveImagePng :: FilePath -> [Double] -> IO ()
saveImagePng path pixels = do
  let w = 28
      h = 28
      clamp x
        | x < 0 = 0
        | x > 255 = 255
        | otherwise = x
      toPx v = fromIntegral (clamp (round (v * 255 :: Double))) :: Pixel8
      idx x y = y * w + x
      pixelAt x y = toPx (pixels !! idx x y)
      img = generateImage pixelAt w h :: Image Pixel8
  createDirectoryIfMissing True (takeDirectory path)
  savePngImage path (ImageY8 img)

printAsciiImage :: [Double] -> IO ()
printAsciiImage pixels = do
  let w = 28
      h = 28
      chars = " .:-=+*#%@"
      n = length chars
      charFor v =
        let i = floor (v * fromIntegral (n - 1))
         in chars !! i
      idx x y = y * w + x
      row y = [ charFor (pixels !! idx x y) | x <- [0 .. w-1] ]
  mapM_ (putStrLn . concatMap (:[])) [row y | y <- [0 .. h-1]]
