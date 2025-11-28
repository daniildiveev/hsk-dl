module TensorSpec (spec) where

import Control.Monad (zipWithM_)
import Core.Tensor
import Test.Hspec

spec :: Spec
spec = describe "Core.Tensor (autograd + basic ops)" $ do
  it "adds tensors elementwise" $ do
    a <- fromList [2, 2] [1, 2, 3, 4]
    b <- fromList [2, 2] [10, 20, 30, 40]
    c <- add a b
    values c `shouldBe` [11, 22, 33, 44]

  it "computes gradients for sum(x * y)" $ do
    x <- fromListWithGrad [2] [1, 2] True
    y <- fromListWithGrad [2] [3, 4] True
    z <- mul x y
    s <- sumAll z
    backward s
    gradValues x `shouldReturn` [3, 4]
    gradValues y `shouldReturn` [1, 2]

  it "runs matmul and backprop" $ do
    a <- fromListWithGrad [2, 3] [1, 2, 3, 4, 5, 6] True
    b <- fromListWithGrad [3, 1] [1, 0, -1] True
    out <- matmul a b
    values out `shouldBe` [-2, -2]
    s <- sumAll out
    backward s
    gradValues b `shouldReturn` [5, 7, 9]
    gradValues a `shouldReturn` [1, 0, -1, 1, 0, -1]

  it "softmax rows sum to one" $ do
    t <- fromList [2, 2] [0, 0, 0, 2]
    probs <- softmax t
    let rows = chunksOf 2 (values probs)
    map sum rows `shouldSatisfy` all (\s -> abs (s - 1) < 1e-9)
    values probs `shouldBeCloseTo` [0.5, 0.5, exp 0 / (exp 0 + exp 2), exp 2 / (exp 0 + exp 2)]

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs =
  let (h, rest) = splitAt n xs
   in h : chunksOf n rest

shouldBeCloseTo :: [Double] -> [Double] -> Expectation
shouldBeCloseTo xs ys = do
  length xs `shouldBe` length ys
  zipWithM_ (\a b -> abs (a - b) `shouldSatisfy` (< 1e-6)) xs ys
