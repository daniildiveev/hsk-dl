module Main where

import qualified GradlessSpec
import qualified ShapeSpec
import qualified TensorSpec
import Test.Hspec

main :: IO ()
main = hspec $ do
  describe "Stage 0 skeleton" $ pure ()
  TensorSpec.spec
  ShapeSpec.spec
  GradlessSpec.spec
