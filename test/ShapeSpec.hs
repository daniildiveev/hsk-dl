module ShapeSpec (spec) where

import Core.Shape
import Test.Hspec

spec :: Spec
spec = describe "Core.Shape helpers" $ do
  it "normalizes negative axis" $ do
    normalizeAxis (-1) [2, 3, 4] `shouldBe` Right 2
    normalizeAxis 1 [2, 3, 4] `shouldBe` Right 1
    normalizeAxis 4 [2, 3] `shouldSatisfy` isLeft

  it "infers reshape with -1" $ do
    inferReshape [2, 3, 4] [2, -1] `shouldBe` Right [2, 12]
    inferReshape [2, 2] [3, -1] `shouldSatisfy` isLeft

  it "broadcasts shapes" $ do
    broadcastShapes [3, 1, 5] [1, 4, 5] `shouldBe` Right [5, 4, 3]
    broadcastShapes [2, 2] [3] `shouldSatisfy` isLeft

isLeft :: Either a b -> Bool
isLeft (Left _) = True
isLeft _ = False
