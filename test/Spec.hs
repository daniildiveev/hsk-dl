module Main where

import Test.Hspec
import qualified TensorSpec
import qualified ShapeSpec
import qualified GradlessSpec

main :: IO ()
main = hspec $ do
  describe "Stage 0 skeleton" $ pure ()
  TensorSpec.spec
  ShapeSpec.spec
  GradlessSpec.spec
