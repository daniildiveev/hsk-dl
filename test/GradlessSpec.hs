module GradlessSpec (spec) where

import Core.Tensor
import Test.Hspec

spec :: Spec
spec = describe "Functional ops without grad" $ do
  it "relu works without grad" $ do
    t <- fromList [4] [-1, 0, 1, 2]
    r <- relu t
    values r `shouldBe` [0, 0, 1, 2]

  it "imageToTensor builds a 2D tensor" $ do
    img <- imageToTensor ([[1, 2], [3, 4]] :: [[Double]])
    shape img `shouldBe` [2, 2]
    values img `shouldBe` [1, 2, 3, 4]
